# Missing features — R&D-scale GreenBoost gaps found 2026-07-26

## How to read this

Written during a gb-quant audit of the LongLive-2.0-5B video pipeline
(`~/.claude-accounts/ferran.duarri-me.com/plans/you-were-implementing-this-reflective-clock.md`).
Each item below is a real, verified gap in what GreenBoost offers for local AI
inference, found by cross-referencing this repo against three external
reference trees the user pointed at: `~/Dev/greenboost_all/gb-quant-sources`
(16 pristine upstream clones — cutlass, TensorRT-LLM, triton, auto-round,
LMCache, tquant2, and others) and
`~/Dev/greenboost_all/greenboost-quants-techs-github/gemlite` (confirmed
redundant as a capability source — same upstream commit as what's already
vendored at `third_party/gemlite`, kept only as an upstream-diff baseline;
see its own `NOTICE` file).

Every item is **documented, not implemented, except (a)**, per an explicit
user decision this session ("implement the safe wins now, document the rest,
but actually run the one cheap experiment"). Each entry gives concrete
file:line pointers (verified against the actual source trees, not just
agent-reported) and an honest effort estimate.

**This file is a living backlog, not a closed list** — a separate, parallel
session (`plans/bring-gb-synapse-gb-quant-and-async-nygaard.md`) is expected
to append its own deferred-item entries here (its P3/P8/P9 phases) rather
than duplicate this file. If you're picking this up fresh, check for
sections below that don't match the "(a)-(g)" numbering from this initial
pass — they were added later.

---

## What was actually implemented and verified this session (not an R&D item — landed, working)

The parent plan's Part 1 (text-encoder gb-quant bridge) and Part 2
(per-phase dataflux telemetry) both landed and were confirmed end-to-end with
a real render: `workers/video/gen_longlive.py --gb-quant near_lossless`
produced a valid 1280x704 h264 mp4, all 4 telemetry phases
(`longlive:build_dataset`/`:inference`/`:encode`/`:render`) appeared
correctly in `dataflux_summary`'s `stages` table (previously invisible under
the old `video_render` kind), and `dataflux_models` now shows
`LongLive2.0-5B-NVFP4-S4` for the first time.

**One honest, unflattering finding worth recording plainly**: this
gb-quant-enabled render's inference phase took **1400.85s**, vs. an earlier
same-day baseline render (no gb-quant) at **1023.94s** — about **37%
slower**, not faster. This matches what the micro-harness
(`tools/bench/longlive_te_quant_ab.py`) found in isolation: quantized
encoding of 8 short prompts took 44.7s vs. 19.6s bf16 — the fp8 GEMM path's
per-call overhead (plus the first-call sensitivity-calibration cost) doesn't
pay for itself at this workload's shape (single-prompt calls, small batch,
one render). **The real win this session was footprint, not speed**: `T1
1.0 GiB + T2 3.5 GiB` (via shim) vs. the naive bf16 load sitting entirely in
T1 (~10.5 GB), and — notably — this specific render needed **zero T2
spillover at all** (`t2_peak_mb: 0` in the render's own telemetry), whereas
earlier same-session renders without gb-quant did spill several GB to T2.
So: gb-quant traded wall-clock time for headroom in this configuration.
Whether that's a good trade depends on what the freed VRAM is used for next
(e.g. a longer multi-shot render that would otherwise spill more
aggressively) — untested this session, since multi-shot validation
(`total_chunks > 1`) is explicitly out of scope for the parent plan.

**Per the user's standing quant-precision policy** ("fp8 default; nvfp4
only if faster AND equal-or-better quality; int4 never default") — this
result does NOT justify adopting nvfp4 for this component even if item (a)
below turns out to unblock it: fp8 itself already isn't a wall-clock win
here, so nvfp4 (which the policy requires to beat fp8 on BOTH axes) has an
even higher bar to clear. Treat gb-quant on this text encoder as a
footprint tool for VRAM-constrained scenarios, not a default-on speed
optimization, until a workload shape (longer batches, more concurrent
prompts) shows the fp8 GEMM overhead amortizing.

---

## (a) NVFP4 GEMM on sm_120 (Blackwell) — TESTED THIS SESSION

**Status:** see the isolated experiment result at the bottom of this section
— run live, not just documented, per the user's explicit request.

**Background.** `gb_quant.py`'s NVFP4 path uses GemLite's Triton kernels
(`_init_nvfp4` at `gb_quant.py:496` originally; line numbers may have moved —
grep for `_init_nvfp4`). NVFP4 is documented as blocked on this box's Triton
by a sm_120 compiler crash (`workflow/gb-quant.md`'s NVFP4 section,
`workflow/known-issues.md`). `gpu_profile()` on this box returns
`quality_default='nvfp4'` for the Blackwell family (verified live this
session), so any quality-tier gb_quant call risks hitting this crash unless
the caller strips nvfp4 from the ladder first (which the LongLive bridge —
`ai-forge/envs/longlive/gb_quant_bridge.py` — now does defensively).

**The fix candidate, found in `gb-quant-sources/triton`:** commit
`d65880ebf2`, "[NVIDIA] Fixed mixed-prec scaled-dot lowering for sm120
(#9577)" — *"we are incorrectly trying to lower mixed-prec workload to
native mxfp MMAv2, which is currently not supported. Fixed by falling back
to the decomposition path instead."* Confirmed via `git merge-base
--is-ancestor` in that clone: **present** in `origin/release/3.8.x` and
`origin/main`, **absent** from `origin/release/3.7.x`. This box's Triton
(checked via the `synapse-torch-env`/`artpipeline_cu13` envs at audit time)
is 3.7.x.

**Two other things worth checking alongside the version bump, both found
live this session:**
1. `/usr/local/cuda-13/bin/ptxas` (→ cuda-13.3) exists on this box now — an
   earlier `known-issues.md` note claiming no CUDA 13 ptxas was present is
   **stale**. `gb_quant._init_nvfp4`-equivalent logic probes exactly this
   path for `TRITON_PTXAS_BLACKWELL_PATH`. Since NVFP4 quantization itself
   (weight packing) already works — it's specifically the GEMM *compile*
   that crashes — this ptxas availability may independently matter and
   should be re-confirmed, not assumed still-missing.
2. PyPI has no Triton 3.8.x wheel; the nightly PyTorch wheel index
   (`https://download.pytorch.org/whl/nightly/`) does, matching this box's
   Python/CUDA ABI (cp312, cu128/cu130 depending on which env).

**Experiment procedure (isolated — this box runs GreenBoost's own production
serving, so nothing touches the real installed Triton):**
- Run a control smoke test under the **currently installed** Triton first —
  honest baseline, since the ptxas availability alone might already have
  changed the outcome independent of any Triton version bump.
- Layer a `venv --system-site-packages` on top of the real interpreter (venv
  site-packages precede system ones on `sys.path`, so a new Triton shadows
  the old one without a single write inside the real env) and `pip install
  --no-deps` the matching 3.8.x nightly wheel.
- Re-run the identical smoke test under the shadowed Triton.
- Teardown: `rm -rf` the venv, re-verify the real env's Triton version is
  unchanged.
- Smoke test itself: `gb_quant._init_nvfp4()` (sets the ptxas path env var),
  build a plain `nn.Linear`, quantize it to nvfp4 via `gb_quant.quantize_module`,
  then actually **call** the quantized layer (`lin(x)`) — quantization is
  known to work; the GEMM call is where it previously crashed. Compare
  output against a bf16 reference (cosine similarity) and time it against
  the fp8 path, since per the user's standing quant-precision policy
  ("fp8 is the default; nvfp4 only if faster AND equal-or-better quality"),
  "it compiles" is not sufficient grounds to adopt it — it must beat fp8.

**RESULT: experiment run, confirmed and complete.** Actual installed Triton in
the `longlive2_nvfp4` env (the env used throughout this session, not
`artpipeline_cu13`) is **3.6.0**, not 3.7.x — corrected from the assumption
above; `d65880ebf2` is absent from `release/3.6.x` too, so the blocker applies
identically.

Control (installed Triton 3.6.0):
```
$ python nvfp4_gemm_smoke.py
NVFP4_GEMM_RESULT: FAIL RuntimeError: PassManager::run failed
  ... gemlite/triton_kernels/gemm_splitK_kernels.py:804 gemm_splitK_forward
  ... triton/backends/nvidia/compiler.py:378 make_llir
  RuntimeError: PassManager::run failed
```
Reproduces the documented crash exactly — confirms the blocker is still live
before touching anything.

Isolated venv (`python -m venv --system-site-packages`, layered on the real
`longlive2_nvfp4` interpreter, zero writes to the real env — verified before
and after), `pip install --no-deps --index-url
https://download.pytorch.org/whl/nightly/ triton==3.8.0+git43422b04`:
```
$ ./triton38_venv/bin/python -c "import triton; print(triton.__version__, triton.__file__)"
3.8.0 .../triton38_venv/lib/python3.12/site-packages/triton/__init__.py
```
First attempt hit an unrelated environment artifact (`torch._inductor`'s
async-compile subprocess pool: `RuntimeError: Could not find an active GPU
backend`, from `quantize_nvfp4`'s own `torch.compile`'d packing step, not the
GEMM). Setting `TORCHINDUCTOR_COMPILE_THREADS=1` (forces in-process compile,
sidesteps the subprocess-pool quirk) resolved that; **the NVFP4 GEMM itself
then compiled and ran successfully**:
```
$ TORCHINDUCTOR_COMPILE_THREADS=1 ./triton38_venv/bin/python nvfp4_gemm_smoke.py
NVFP4_GEMM_RESULT: OK shape=(16, 4096) dtype=torch.bfloat16 cosine_vs_bf16=0.99489 wall_s=0.000088
FP8_GEMM_RESULT: OK cosine_vs_bf16=0.99967 wall_s=0.000080
```
(second run, with a warmup call added before the timed loop so first-call
compile cost isn't smeared into the per-call average — the first, unwarmed
run showed ~0.30s for both paths, which was pure compile overhead, not GEMM
cost, and is not a valid comparison).

Teardown: `rm -rf triton38_venv`; re-verified real env unchanged:
```
$ python -c "import triton; print(triton.__version__, triton.__file__)"
3.6.0 /home/ferran/.miniforge3/envs/longlive2_nvfp4/lib/python3.12/site-packages/triton/__init__.py
```

**Verdict against the user's standing policy ("fp8 default; nvfp4 only if
faster AND equal-or-better quality; int4 never a new default"): fp8 wins on
both axes.** fp8 is ~9% faster (80µs vs 88µs per call, this shape) and
higher-fidelity (cosine 0.99967 vs 0.99489) than nvfp4, even with the Triton
3.8.x fix applied. **Conclusion: bumping Triton to 3.8.x does unblock NVFP4
compilation on this box (a real, reproducible fix — worth doing if NVFP4 is
ever needed for a VRAM-fit reason gb-quant can't solve with fp8 alone), but it
does not change the policy outcome — fp8 remains the correct default for this
card, this shape, today.** This is a single 4096×4096 linear at batch 16, one
shape point, not a sweep — if a future workload is VRAM-constrained enough
that fp8 doesn't fit but nvfp4 would, re-run this smoke test at that model's
actual shapes before deciding, since the crossover point (if any) is
shape-dependent and unmeasured here. Upgrading the real env's Triton to
3.8.x-nightly is NOT recommended as a standing change purely for this result — it
would need its own validation pass (it's a pre-release/nightly build, and
the `TORCHINDUCTOR_COMPILE_THREADS=1` workaround needed above is itself a
sign of rough edges) against a benefit that this test didn't find.

---

## (b) Non-Triton CUTLASS NVFP4×BF16 weight-only GEMM

GemLite's NVFP4 path is 100% Triton, which is what (a) is blocked on. A
non-Triton CUTLASS kernel would be immune to that specific compiler crash
class entirely.

**Source:** `gb-quant-sources/cutlass/examples/79_blackwell_geforce_gemm/`
has four example variants — `79a_blackwell_geforce_nvfp4_bf16_gemm.cu` is
the **weight-only NVFP4×BF16** one, matching GreenBoost's actual usage
pattern (`gb_quant.py` only ever builds weight-only `A16W*` processors, per
its own precision ladder — activations stay BF16). `79b` is NVFP4×NVFP4
(both operands quantized — not our shape), `79c` is mixed MXFP8/MXFP6/BF16,
`79d` is grouped GEMM (MoE-shaped). All four confirmed present.

**Production-grade dispatch reference:**
`gb-quant-sources/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h`
— confirmed present alongside sibling `_sm100.h` and a shared
`fp4_gemm_template.h`. This solves the tile-config/dispatch/workspace
plumbing that the bare CUTLASS example leaves to the integrator.

**Checked, not directly usable:** the `tilegym-adding-cutile-kernel` Claude
Code skill (`~/Dev/greenboost_all/nvidia_skills/skills/skills/`, installed
this session to `~/.claude/skills/`) documents NVIDIA's own end-to-end
process for adding a cuTile operator — but it's specific to TileGym's own
architecture (`@dispatch` in `src/tilegym/ops/ops.py`, its own
`__init__.py`/benchmark conventions), not transferable line-for-line to
`gb_kernel_backends.py`'s GemLite-based `_build_processor`/`from_linear`
pattern. Worth a skim for the general shape of "register a new kernel
backend + dispatch by (bits, shape, arch) + test + benchmark" if/when this
item is actually implemented, not a drop-in guide.

**Lands at:** `gb_kernel_backends.py`'s `build_cutlass_nvfp4_processor`
(currently a stub returning `None` — confirmed at the time this doc was
written; re-check the line number, it will have moved) and the existing
scaffold `tests/bench/bench_cutlass_nvfp4.py`, whose own docstring already
flags the operand construction as a placeholder.

**Effort: 1-3 days.** The hard part is the operand-layout repack —
`gb_quant`'s NVFP4 packing layout needs reconciling with CUTLASS's own
interleaved scale-factor layout; the kernel call itself is comparatively
straightforward once that's done. Genuine CUDA/C++ integration work, not a
config change.

---

## (c) DP/budget-optimal mixed-precision layer planner

`gb_quant.py`'s `plan_quality` walks the precision ladder per layer and
greedily picks a precision — it does not solve a global budget-constrained
optimization, so a low-sensitivity layer can't be traded down to buy a
high-sensitivity layer more precision within a fixed total budget.

**Source:**
`gb-quant-sources/auto-round/auto_round/auto_scheme/delta_loss.py:1214
choose_bits_per_layer_with_path(layers, P, max_states)` — confirmed present
at that exact line. Budget-constrained dynamic program over
`(scheme, bits_cost, loss_cost)` per layer, with Pareto-pruning of dominated
`(params, loss)` states and an explicit beam-width cap (the upstream
comment notes that without it, distinct cumulative-bit sums can blow past
70 GB of RAM during search — a real scaling concern to carry over, not
just copy the algorithm).

**Design:** an opt-in `GB_QUANT_DP_PLAN=1` alternative path alongside the
existing greedy `plan_quality`, feeding it `gb_quant_calib`'s existing
per-layer sensitivity measurements as the loss-cost input. Additive, not a
replacement — the existing calibration-free zero-data proxy path stays the
default (it's why calibration works at all under the LD_PRELOAD shim with
no dataset, even on a small feeder box).

**Effort: ~1 day** — mostly adapting the tuple shape to
`gb_quant_calib.py`'s existing `{layer: {bits: rel_err}}` sensitivity dict.

---

## (d) Diffusion/video-aware calibration

`gb_quant_calib.py`'s sensitivity measurement is weight-only: quantize a
layer's weight tensor, dequantize, measure Frobenius relative error — never
an actual forward pass through the model, and never diffusion-loop-aware.
For a 24-layer causal video transformer, this per-layer weight-error proxy
and the true end-to-end output drift can differ substantially (confirmed
empirically this session: gb_quant reported `mean_err=0.0264` at the weight
level for a fp8-quantized text encoder, while the actual end-to-end
embedding cosine similarity/relative error measured across a real forward
pass was `cos≈0.997`, `rel_err≈0.08` — expected, since error compounds
through nonlinear layers, but it means the weight-level number alone
understates what an activation-aware calibration would actually measure).

**Source:**
`gb-quant-sources/auto-round/auto_round/calibration/diffusion.py` (a
`DiffusionCalibrator` that drives the real `pipe` through actual denoising
steps, including I2V calibration-image synthesis) and
`gb-quant-sources/auto-round/auto_round/compressors/diffusion_mixin.py`
(explicit **WAN dual-transformer** handling — directly relevant, since
LongLive is Wan2.2-based). Both confirmed present.

**Effort: 1-2 days.**

---

## (e) KV-cache tier-serde compression — the item that would actually help LongLive-class models

This is the highest-value item on this list for causal/flash-attn video
models specifically, and it's worth being precise about *why* the existing
mechanism doesn't cover this case (confirmed by reading LongLive's model
code directly, not assumed): `gb_attn.py`'s TurboQuant K/V compression works
by monkey-patching `torch.nn.functional.scaled_dot_product_attention`
globally, and even then only for non-causal calls with no explicit
attn_mask. LongLive's causal transformer calls flash-attn varlen kernels and
`torch.nn.attention.flex_attention` directly — never `F.sdpa` — so
`gb_attn`'s patch is architecturally unreachable for this model family, not
merely unconfigured. Forcing it on would be an inert no-op at best.

The actually-reachable mechanism is different: compress KV **at the T2/T3
tier-move boundary** (`gb_model_tier.py`'s existing spill path), not inside
the attention call itself. This works for ANY model regardless of its
attention implementation, because it operates below the model's own code
entirely — on whatever bytes get spilled to DDR/NVMe when the shim's
overflow logic decides a KV allocation doesn't fit in T1.

**Source:**
`gb-quant-sources/LMCache/lmcache/v1/distributed/serde/turboquant/`
(4 files confirmed present: `__init__.py`, `turboquant.py`, `store_kernel.py`,
`decode_kernel.py` — TurboQuant reimagined as a storage-tier serde: compress
on store, decode on prefetch, with named presets like `turboquant_k8v4`
including a norm-correction flag GreenBoost's own gb_attn doesn't have) plus
the CacheGen GPU arithmetic coder
(`gb-quant-sources/LMCache/csrc/ac_enc.cu`, `ac_dec.cu`, `cal_cdf.cu` — all
confirmed present) for lossless entropy compression stacked on top of the
quantization.

**Effort: 2-3 days** (kernel work + integration with `gb_model_tier.py`'s
existing T2/T3 spill decision path). Flag as the item most worth doing next
if this pipeline's KV-cache footprint on T2/T3 becomes a real bottleneck at
longer render durations (multi-shot, `total_chunks > 1` — not yet validated
this session, out of scope per the parent plan).

---

## (f) Persistent video-serving engine

Every LongLive render currently pays the full model-load cost (5B NVFP4
transformer + VAE + UMT5-XXL text encoder) as a fresh subprocess — measured
this session at **~17 minutes for a single 1.2-second, 1-shot clip**, most
of which is one-time load/quantization cost rather than actual diffusion
compute. `workers/movie/orchestrate.py` batches many scenes sequentially, so
this cost is paid once per scene rather than once per pipeline invocation.

**Existing precedent that's close but doesn't fit:**
`gb_synapse_backends.DiffusersBackend` / `gb_diffusion_server.py` already
keeps a diffusion pipeline resident across requests — but it's hardcoded to
`AutoPipelineForText2Image` (image-only) and exposes
`POST /v1/images/generations` (prompt/n/size/steps → base64 PNGs). LongLive
is a causal, multi-shot, dataset-driven video pipeline with no `AutoPipeline`
entry point at all — structurally incompatible with reusing that server as-is.

**Design shape:** a new `gb_longlive_server.py` (same "load once, serve many"
pattern as `gb_diffusion_server.py`) + a new `"video"`/`"longlive"` engine in
`gb_synapse_backends.select_backend`. Requests would carry a shot list +
i2v keyframe path rather than a single prompt.

**Effort: 1-2 days.**

---

## (g) Per-PID shim telemetry

`/run/greenboost/shim_stats` and the accompanying `metrics.json` are single
global files, not per-PID. This created a real, documented ambiguity during
this session's earlier debugging: it was genuinely unclear at one point
whether a read of `shim_stats` reflected the process actually under
investigation or a stale snapshot left by a different process that had
touched the same global file moments earlier.

**Fix would require:** a `greenboost_cuda_shim.c` change — e.g. writing to
`/run/greenboost/shim_stats.<pid>` (or an equivalent per-process path) — plus
matching updates to every reader (`gb_monitor.py`'s `read_shim_stats`/
`parse_shim_stats`, and anywhere else that hand-parses the current global
path).

**Effort: unknown without reading the shim's stats-write path first.**
C-level change; explicitly out of scope for a Python-only session. Flagged
here so it doesn't get lost.

---

## (h) Preflight memory-sizing query for gb-quant (minor, low-effort)

Surveyed `~/Dev/greenboost_all/nvidia-dlss` (NVIDIA's DLSS + NVIDIA Image
Scaling SDKs) on request, for anything reusable. **Most of it doesn't
transfer**: DLSS itself is closed-source (only the public C header API for
game integration is available — the actual upscaling network is a
proprietary DLL, nothing algorithmic to borrow), and NIS's open-source
shader code operates on rendered game frames, a completely different data
domain from GreenBoost's model weights/KV cache tensors.

**One genuinely transferable pattern, though:** DLSS's
`NVSDK_NGX_{D3D11,D3D12,CUDA}_GetScratchBufferSize` API
(`DLSS/include/nvsdk_ngx.h`, confirmed present) — a dedicated **preflight
query** returning the exact scratch-memory an operation will need, called
*before* allocating anything, rather than allocating heuristically and
correcting after the fact. `gb_quant._auto_budgets()` today estimates
T1/T2 budgets heuristically (a fraction of `torch.cuda.mem_get_info()` plus
the T2 pool's advertised size) rather than asking "how much will THIS
specific quantization plan actually need." A `plan_fit`/`plan_quality` dry
run that returns a precise byte count *before* any GPU allocation happens
(rather than the current after-the-fact `quantize_to_fit`'s live-telemetry
tightening) would be a small, low-risk refinement in the same spirit.

Also noted in passing: DLSS's `NVSDK_NGX_PerfQuality_Value` enum
(`MaxPerf`, `Balanced`, `MaxQuality`, plus `Ultra*` variants) is the same
*shape* as `gb_quant.QUALITY_TIERS` — named quality/performance tiers
instead of raw numeric knobs — just with finer granularity (5-6 tiers vs.
gb_quant's 3). Not something to copy directly (GreenBoost's 3 tiers map
cleanly to fp8/mixed/floor and adding more doesn't obviously help), but
worth knowing the precedent exists if finer-grained tiers are ever
requested.

**Follow-up pass (deeper agent survey, same request, re-confirmed against the
actual tree, not just the headers):** NVIDIAImageScaling (NIS) is a genuinely
separate, MIT-licensed, open-source SDK bundled alongside DLSS in the same
directory — its shader math (`NIS_Main.hlsl`/`.glsl`) is irrelevant (rendered
game frames, not model tensors), but its **CPU-side config code** has two
more concrete shape-level patterns worth citing here rather than copying
verbatim:
- `NISOptimizer` (`NVIDIAImageScaling/NIS/NIS_Config.h:91-153`): a tiny
  `enum class NISGPUArchitecture` → `constexpr` switch table returning tuned
  numeric knobs (block width/height, thread-group size) per architecture
  tier. Same *shape* as a hardware-tier-keyed dispatch table; relevant to
  `gb_telemetry.py`'s topology detection and `gb_tiering.py`'s budget
  selection **as a pattern to imitate, not literal values to copy** (a fixed
  constexpr table is itself in tension with this repo's own
  never-hardcode-a-hardware-value rule if used naively — see the anti-pattern
  below).
- `NVScalerUpdateConfig()` (`NVIDIAImageScaling/NIS/NIS_Config.h:156-254`): a
  single scalar "sharpness" knob (0..1) mapped through split ranges with
  different scale-limit constants per range, deriving several dependent
  parameters, with explicit input validation (rejects out-of-range scale
  ratios). Same *shape* as a single quant-aggressiveness dial feeding several
  dependent precision/threshold parameters — worth studying the
  interpolation/validation style if `gb_quant.py` ever grows a continuous
  (rather than discrete-tier) quality knob.
- **A concrete anti-pattern worth citing, not copying**:
  `NVIDIAImageScaling/samples/DX11/src/DeviceResources.cpp:55-58` correctly
  queries the real adapter's `DXGI_ADAPTER_DESC.VendorId` — but
  `NVIDIAImageScaling/samples/DX11/src/NVScaler.cpp:39` never consults that
  queried VendorId and hardcodes `NISGPUArchitecture::NVIDIA_Generic`
  regardless. This is exactly the "hardcoded hardware value despite having
  the real detection available two files away" failure mode this project's
  CLAUDE.md prohibits — a useful negative example if this document is ever
  used to onboard someone to that rule.

**Effort: ~half a day** (a dry-run mode for `plan_fit`, no new kernel work) —
by far the smallest item in this document; genuinely candidate for a future
quick win rather than R&D-scale like (a)-(g). The NIS config-code patterns
above are cited for their shape only; no further implementation effort is
proposed for them this round — DLSS/NIS survey is complete and closed as of
this pass.

---

## Appendix A — why `gb_attn.py`'s TurboQuant is unreachable for LongLive (and models like it)

Verified directly against the model source, not assumed: LongLive's causal
transformer calls flash-attn varlen kernels and `flex_attention` directly
(never `F.scaled_dot_product_attention`), and even LongLive's VAE (the one
place that DOES call plain SDPA) isn't a meaningful target — quantizing K/V
there buys negligible memory since VAE attention is a small fraction of the
model's total footprint. Any model built the same way (flash-attn-direct,
not `F.sdpa`-mediated) will have this same gap. See item (e) for the actual
applicable mechanism.

## Appendix B — why the LongLive transformer and VAE are not gb-quant weight targets

- The 5B causal transformer ships already pre-quantized NVFP4 by the
  LongLive authors (`FourOverSixLinear(nn.Linear)` in the vendored
  `fouroversix` package) — re-quantizing it would be redundant at best,
  corrupting/crashing at worst (which is why `gb_quant.py` now has an
  `is_prequantized_linear()` guard, landed this session, specifically
  because this class subclasses `nn.Linear` and would otherwise match
  `_delegate_patch`'s naive `isinstance` scan).
- The VAE is `nn.Conv3d`-heavy; `gb_quant.py` only ever touches `nn.Linear`
  layers, so quantizing it yields negligible footprint reduction — and it's
  already on gb_quant's own default skip-component list for pipe-shaped
  callers.
- The one real, safe weight target in this pipeline is the UMT5-XXL text
  encoder — see the parent plan's Part 1 for the bridge that now handles it
  (`ai-forge/envs/longlive/gb_quant_bridge.py`).

## Appendix C — a real, reproducible GreenBoost timing race (found and worked around this session, not yet root-caused)

While building the text-encoder A/B micro-harness
(`ai-forge/tools/bench/longlive_te_quant_ab.py`), the SAME code
(building `WanTextEncoder()` then running one forward pass) failed with a
genuine `torch.OutOfMemoryError` on some runs and succeeded cleanly on
others — including one successful run with LESS free VRAM headroom than a
failing run moments before it. This is consistent with (though not fully
proven to be) the allocation-cadence-sensitive race documented earlier the
same day: `gb_needs_overflow()`'s cached `cuMemGetInfo` value can go stale
during a rapid burst of many small allocations (loading a multi-GB encoder
parameter-by-parameter), letting physical VRAM get walked past the shim's
own reserve before any single allocation is large enough to trigger the
overflow-to-T2 decision. `GREENBOOST_DEBUG=1`'s added per-call logging
overhead — which slows allocation issuance — correlated with (but did not
reliably guarantee) success across repeated attempts. **This needs the same
kind of precise, correlated-timeline diagnosis as the earlier investigation
that day, not another guess** — a good next step would be instrumenting
`gb_needs_overflow`'s cache-staleness window directly (e.g. a shadow byte
counter incremented on every T1 allocation since the last real
`cuMemGetInfo` refresh, subtracted from the cached free-VRAM estimate) rather
than continuing to work around it with debug-logging overhead as a crutch.

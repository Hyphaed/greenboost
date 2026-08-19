# tasks_10x.md — execution breakdown for `plan_10x.md`

Written 2026-08-17. Tasks are ordered; `W0` gates everything after it.
Each task states **Done when** so completion is not a judgement call.

Legend — `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## P — MEASURED PRIORITIES (do these first; supersede W0-W3)

W0 is complete — see `plan_10x.md` §0.1. Baseline is **17.8 tok/s latest /
33.2 avg**, MTP speculation already runs, and `dataflux_critic` names the real
bottleneck. These two tasks are worth more than everything below combined.

### [~] P1 — RESEARCHED 2026-08-17: the alert was wrong, the regression is worse

**The "46% regression vs avg 33.2" was a false alarm built on a contaminated
mean.** Real distribution over 2 days on the reference workload:

| ctx | n | **median** | mean | min | max |
|---|---|---|---|---|---|
| 32768 | 41 | **13.9** | 33.2 | 1.7 | 283.8 |
| 65536 (current serve) | 2 | **3.4** | 3.4 | 3.4 | 3.5 |
| *(unkeyed)* | 1 | — | **21065.2** | | |

Mean sat **2.4x above median** — so a sample near normal read as a crash.

**But the real regression is bigger than the alert claimed, and elsewhere.**
The current ctx=65536 serve runs at **3.4 tok/s against a 13.9 median** (~75%
down), and the rollup missed it entirely because ctx=65536 is a *different key*
from the ctx=32768 history the alert compared against.

Chain (consistent with P2a): ctx 32768 → 65536 ⇒ KV reserve formula scaled to
5628 MB ⇒ VRAM fill 45-50% ⇒ more weights streaming from T2 over PCIe every
forward pass ⇒ decode collapses. **The P2a KV fix (1322 MB, −2.9 GB reserve)
targets exactly this**, so re-measure after a restart before doing anything else.

**Three telemetry defects found and fixed (this is why it was invisible):**
1. **Sample-quality floor applied to one path only.** Streaming enforced
   `_MIN_TOK_S_SAMPLE_TOKENS` (24); the non-streaming branch — taught to record
   in 2026-08-05 — only checked `> 1`, so 2-token replies became full-weight
   samples scoring hundreds of tok/s. **That is the measured source of the
   254.9/283.8 outliers.** Floor now applied to both.
2. **Samples were duration-blind.** `duration_s` was `0.0` on every event and no
   token count was recorded, so a 3-token reply and a 500-token generation were
   indistinguishable. `completion_tokens` now threaded
   `gb_synapse_api → record_measured_tok_s → _df_emit_tok_s` and emitted.
3. **Rollup published only `avg`.** `summarize()` now also publishes
   `median`/`min`/`max` (`avg` kept for compatibility). Verified live: the
   ctx=32768 key now reports `median=13.9` beside `avg=33.2`.

Tests: 2 new in `test_gb_synapse_api_tok_s.py`; 4 over-specified stubs there
widened to tolerate additive kwargs. **Full suite: 2071 passed, 1 skipped, 0 failed.**

**Next:** make the pilot's regression detector compare against `median`, not
`avg` — the fix above publishes the number but the detector still reads `avg`.

### [ ] P1-orig — Recover the 46% decode regression (superseded by the above)
`dataflux_tok_s` shows latest 17.8 vs avg 33.2 over 41 samples. The pilot warns:
*"Check for a competing GPU load or KV spill to T2 … [lever: set_kv_size_threshold_mb(?)]"*.
```
mcp__greenboost-dataflux__dataflux_tok_s(model="<full key>")   # regression over time
mcp__greenboost-dataflux__tiering_status()                     # kv_t2_mb, t2_pressure
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # competing load?
```
Note ctx=65536 with `kv_reserve_mb=1877`; a larger ctx than needed inflates the
KV reserve and steals VRAM from weights.
**Done when:** cause identified and median tok/s is back at or above 33.
**Do not tune anything else until this is understood** — optimising against a
regressed baseline will produce false attributions.

### [x] P2a — ROOT CAUSE FOUND + FIX APPLIED 2026-08-17: missing KV measurement
The underfill is not mysterious. `/run/greenboost/shim_stats` on the live serve:
```
kv_reserve_nominal_mb  = 5628     kv_t1_tracked_mb = 1322   <- 4.3x over-reserved
kv_reserve_effective_mb= 4305     vram_headroom_mb = 244
t2_overflow_total_mb   = 16312    (16.3 GB actively overflowing)
```
`kv_measurements.json` held entries for **Qwen3.6 Fable-Fusion only** — nothing
for Cold-Fusion, which became the reference model *today*. Per
`_persist_kv_measurement()`'s docstring, a first-ever serve of a
`(model, ctx, kv_type)` signature falls back to `estimate_kv_gb()`, which is the
documented ~2.9-4x over-estimate for this hybrid Gated-DeltaNet architecture.
The oversized reserve then blocks `gb_frontload_split_alloc()` from filling VRAM
— the exact mechanism the docstring describes.

**Applied:** persisted the real measurement via the module's own locked writer:
```python
gs._save_kv_measurement("Qwen3.8-27B-Cold-Fusion-MTP-IQ4_XS", 65536, "f16", 1322.0)
```
1322 MB is the flight recorder's 6-hour **peak** (`_peak_kv_used_mb()`), equal to
the live `kv_t1_tracked_mb`, so it covers the high-water mark. Frees **~2.9 GB**
of VRAM reserve on the **next** serve; the running server is untouched.
**Verify on next restart:** `vram_fill_pct` should move well above 50.3% toward
the 85-92% band, and `rule1_underfilled` should stay unmatched *for the right
reason*. Precedent (2026-08-02, same fix, Qwen3.6): VRAM 67-73% → 85.1%,
decode 2.6-4.3 → 5.27 tok/s.

**Generalised 2026-08-17 — two systemic defects found and fixed:**

1. **Why it never self-captured.** `_persist_kv_measurement()` runs at
   `gb_synapse.py:3757`, immediately after the engine reports ready — which is
   exactly when `kv_t1_tracked_mb` is still **0**, because llama.cpp allocates
   KV lazily on the first request, not at load. The `if kv_t1_mb <= 0: return`
   guard then skips the write, so a model that is never cleanly `stop()`ped
   keeps the formula forever. That is precisely what happened to Cold-Fusion.
2. **Cross-model contamination in the fallback.** `_peak_kv_used_mb()` scanned
   *all* `snapshot` events over 6 hours, and snapshots carry **no model field**.
   Right after a reference-model swap that window is still full of the previous
   model's KV — so the fallback would persist the OLD model's footprint under
   the NEW model's key. Same class as the tok_s keying bug
   (missing_features.md item (k)).
   **Fixed:** `_peak_kv_used_mb(window_s, since_ts=None)` now scopes to one
   serve; `since_ts` narrows and never widens the window. Threaded through both
   call sites — `_launch_proxy_and_record` puts `started_ts` in `common`, and
   `stop()` passes `ServerState.started_ts` (the path that matters most, since
   teardown is when the live counter reads 0 and the fallback takes over).

Tests: 2 new in `tests/test_gb_synapse_kv_measure_persist.py`.
**Full suite after all changes: 2069 passed, 1 skipped, 0 failed.**

**Still open:** the lazy-allocation gap itself. A late capture (e.g. after the
first completed turn, where the proxy already records tok_s) would fix it for
models that are killed rather than cleanly stopped. Currently a serve→SIGKILL
cycle still leaves no measurement.

### [x] P2b — FIXED 2026-08-17: tripwire reported `matched: false` when it meant "unknown"
`semantic_segments("rule1_underfilled")` returned `matched: false` while
`vram_fill_pct=50.3` (below_target) and `t2_overflow_active_mb=null`
(provenance: `/run/greenboost/shim_stats stale`). The shim writes stats every
250 ms *from CUDA hooks*, so the file legitimately goes stale whenever inference
is idle — and the tripwire then reads as **healthy** rather than
**indeterminate**. Truth at that moment was a real violation (16.3 GB
overflowing at 50% fill).
**Fix:** a segment with a `null` in its decisive evidence must not return
`matched: false`. Return an indeterminate/unknown verdict (or gate on
freshness explicitly) so an idle shim can never look like a clean bill of health.
This is the same silent-false-negative class CLAUDE.md's GB-Semantics row already
names for `t2_pressure`/`t2_allocated_mb`.

### [ ] P2c — Remaining underfill work (after P2a is verified)
Critic, 32× in one day: *"physical VRAM only 69% full while T2 DDR held
14326 MB — weights that could live in VRAM are crossing PCIe every access;
front-load real VRAM chunks or use the llama.cpp --rpc split so T1 fills first."*
Now: **5718 / 12227 MB used (47%)**, target ~90% (≈11 GB).
- Find why the shim stops promoting at ~47% while `t2_allocated_mb=0` and
  `blackwell_zerocopy` is the active path.
- Check the interaction between `kv_reserve_mb=1877` and the weight budget
  (`weights=15.9GB/10.3GB budget` — the 10.3 GB budget may itself be the cap).
- Reconcile the CLI's "T2 15.9/40.6G" against the kmod's `t2_allocated_mb=0`;
  one of those two numbers is not describing what it appears to.
**Done when:** VRAM occupancy ≥85% during steady decode, with tok/s re-measured.
**Expected payoff:** the largest single lever identified in this whole effort.

### [ ] P3 — Re-baseline after P1+P2, then re-read this plan
Everything in W1-W3 was scoped against wrong numbers. Re-derive what is left
once VRAM is actually full and the regression is gone.
**Done when:** a fresh `dataflux_tok_s` average is recorded here.

---

## W0 — Ground truth (SUPERSEDED — see §P; kept for method)

> **Read `plan_10x.md` §0 first.** The model is `qwen35` — **DENSE**, 65 blocks,
> 4 KV heads, `full_attention_interval=4` (n_kv_layers≈17). A briefly-recorded
> "MoE, 256 experts" claim was read off the retired model in `/opt/models/` and
> has been retracted.

### [x] T0.0 — Record the real model's architecture — DONE 2026-08-17
Resolved from the **served** path, not a glob:
`/var/lib/greenboost/synapse/models/_hf_cache/…Cold-Fusion…MTP-IQ4_XS.gguf` (15.85 GiB)
```
general.architecture=qwen35 (dense) · block_count=65 · head_count=24
head_count_kv=4 · full_attention_interval=4 · file_type=30
```
**Standing rule this task now encodes:** always resolve the model path from the
live serve line / run-state first. Reading `/opt/models/*.gguf` gave a confident,
specific, wrong answer (`qwen35moe`, 256 experts) for a model that is not served.

Nothing downstream is worth building on unmeasured numbers. Every figure in
`plan_10x.md` §1 comes from one `nvidia-smi` reading plus published results.

### [ ] T0.1 — Measure the real baseline tok/s
Serve the default model as it is served today and record decode throughput.

```bash
# capture the exact serve command GreenBoost actually builds
journalctl -u '*synapse*' -n 200 --no-pager | grep -i llama-server
# then a fixed-prompt benchmark, batch 1, greedy, ≥256 output tokens, 3 runs
```
Record: tok/s (median of 3), prompt-eval vs eval split, `-ngl` chosen, ctx size,
KV type, whether MTP was enabled or suppressed.
**Done when:** a reproducible command + median tok/s is written into this file.

### [ ] T0.2 — Confirm the placement really is dense partial offload
Verify the model is *not* taking the `fits_vram` or `is_moe` fast path.
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # during decode
```
Cross-check the `-ngl N/65` split from T0.1 against `gb_synapse_backends.py`'s
branch logic.
**Done when:** the active branch in `LlamaCppBackend.serve()` is named explicitly.

### [ ] T0.3 — Measure actual bytes/token over PCIe
Validate the 248 ms / 6.2 GB estimate rather than trusting the model arithmetic.
```bash
nvidia-smi dmon -s t -d 1        # rx/tx throughput during steady-state decode
```
**Done when:** measured GB/s and inferred bytes/token are recorded; if they
contradict §1's arithmetic, stop and revise the plan.

### [ ] T0.4 — Reproduce the 2026-08-01 segfault on the *current* build
The crash note is from 2026-08-01 against a then-installed `libllama.so`.
Confirm it still reproduces before planning a fix.
```bash
# force the crashing config: dense partial offload + mtp
llama-server -m <model.gguf> -ngl 36 -c 48640 -ctk q8_0 <mtp/spec flags>
```
**Done when:** either a fresh `rc=-11` backtrace is captured, **or** it no
longer crashes — in which case skip to T1.2 and celebrate.

### [ ] T0.5 — Record the installed llama.cpp revision
```bash
$HOME/.local/share/greenboost/synapse/llama-server --version
git -C ~/.unsloth/llama.cpp log -1 --oneline
```
**Done when:** commit SHA + build date recorded here (needed for any upstream bug report).

---

## W1 — Unblock speculation under partial offload (**this is the 10x**)

### [ ] T1.1 — Check whether upstream already fixed it
Cheapest possible path; do this before writing code.
```bash
git -C ~/.unsloth/llama.cpp fetch origin
git -C ~/.unsloth/llama.cpp log --oneline <installed-sha>..origin/master \
  -- src/llama-context.cpp src/llama-graph.cpp | grep -iE "mtp|spec|draft|offload"
```
Also search upstream issues/PRs for `n_ctx` + speculative + partial offload.
**Done when:** a yes/no answer with commit links, recorded here.

### [ ] T1.2 — Localise the NULL deref
Find where the draft `llama_context` is left unconstructed for partial-offload
placement, and which call site invokes `->n_ctx()` on it unconditionally.
```bash
addr2line -e libllama.so.0 -f -C <fault-addr>     # as done 2026-08-01
gdb --args llama-server <crashing config>          # bt at the fault
```
**Done when:** the offending source file + line and the missing-init site are
both identified.

### [ ] T1.3 — Build llama.cpp from source with symbols
Prerequisite for T1.4; needed regardless of which fix route wins.
**Done when:** a locally built `llama-server` reproduces T0.4's crash.

### [ ] T1.4 — Implement the fix
Preferred: construct the draft context correctly for partial-offload placement.
Minimum viable: guard the `->n_ctx()` path and degrade to non-speculative
instead of segfaulting (a crash is never acceptable, even if speculation is
unavailable).
**Done when:** the T0.4 config runs to completion with `mtp=true`.

### [ ] T1.5 — Verify losslessness
Speculative decoding is output-identical when correct. Compare greedy output,
same seed, speculative vs non-speculative, ≥500 tokens.
**Done when:** byte-identical output, or the divergence is root-caused.
**This gates everything after it — a fast wrong answer is worthless.**

### [ ] T1.6 — Measure τ (average acceptance length)
The single number the whole thesis rests on. Papers report τ≈13-14 (SpecExec)
and up to ~30 (SubSpec).
**Done when:** τ recorded on ≥3 workload types (chat, code, long-form).
**If τ < 4, stop and escalate to T2.3** — our merged model may draft poorly.

### [ ] T1.7 — Measure end-to-end speedup vs T0.1
**Done when:** median tok/s recorded; ≥5x = success, ≥9x = stretch met.

### [ ] T1.8 — Re-enable MTP for the partial-offload branch in GreenBoost
Remove the `fits_vram`/`is_moe` restriction in `gb_synapse_backends.py` (~1030)
**only** for configurations proven safe by T1.4/T1.5. Keep the existing gating
as the fallback path; make the new path separately gated and reversible.
**Done when:** GreenBoost serves the default model with speculation enabled and
the comment block is updated to describe the new, true state.

---

## W2 — Tier-aware drafting (the research contribution)

Only start after W1 proves speculation works at all.

### [ ] T2.1 — Read SubSpec's method section properly
`~/…/scratchpad/papers/2509.18344v2.md` is already converted. Extract exactly
how substitute layers are constructed and which layers are chosen.
**Done when:** a one-page method summary exists in this repo.

### [ ] T2.2 — Expose residency state to a drafter
`ModelTierManager` (`gb_model_tier.py:254`) already tracks T1/T2/T3 per entry.
Add a read-only query returning current per-layer tier.
**Done when:** a drafter can ask "which layers are offloaded right now?".

### [ ] T2.3 — Prototype substitute-layer drafting
Build low-bit substitute layers from *offloaded* portions, sharing the resident
layers and KV-cache. Reuse `_quant_int8_slice` (`gb_moe.py:201`) rather than
writing new quantisation.
**Done when:** a draft model exists that consumes no additional VRAM beyond the
measured 1.8 GB headroom.
**This is also the fallback if T1.4 proves the MTP graph unfixable.**

### [ ] T2.4 — Compare τ: MTP heads vs substitute layers
**Done when:** both measured on the same prompts; the better one becomes default.

---

## W3 — Compound gains

### [ ] T3.1 — Quantify the compression→residency headroom
`_lossless_compress_tensor` (`gb_moe.py:255`) already exists. Measure the real
compression ratio on this model's weights and convert it to GB of residency.
**Done when:** "X GB freed → baseline rises from A to B tok/s" is recorded.

### [ ] T3.2 — Re-measure T3 energy under speculation
`arXiv 2508.06978`: NVMe MoE offload costs ~12x per-token energy, and
prefetching cannot mitigate it. Speculation cuts transfer *count* ~10x, which
may change the verdict.
```bash
nvidia-smi --query-gpu=power.draw --format=csv -l 1   # + wall meter if available
```
**Done when:** joules/token measured with and without speculation.

### [ ] T3.3 — Make tier policy chassis-aware
From T3.2: T3 enabled on the mains-powered desktop, demoted on the
battery-powered laptop (HP Omen, Ryzen AI 9 365). Detect chassis as
`_select_fragments()` does in the kernel repo.
**Done when:** policy differs by chassis and is covered by a test.

### [ ] T3.4 — Wire patches 0019/0020 to a real consumer
The compressed dma-buf descriptor is currently an honest no-op — no GreenBoost
buffer is compressed at the C level. Connect it to the compression in T3.1.
**Done when:** measured byte reduction on the wire, not just a descriptor.

---

## W4 — Publication (only after W1 measures something real)

### [ ] T4.1 — Upstream the llama.cpp fix
A diagnosed, reproducible segfault fix in the speculative path for partial
offload. Small, verifiable, benefits everyone running big models on small GPUs.
Include: the T0.4 repro, the T1.2 root cause, the T1.5 losslessness check.
**Done when:** PR opened with a reproducible test case.

### [ ] T4.2 — Write up tier-aware drafting
Only if W2 beats W1 measurably. Claim exactly what was measured, with the
hardware and model stated. Existing publishing tracks are documented in
`~/Dev/kernel_inference/docs/ferran_custom_patches/PUBLISHING.md`.
**Done when:** draft written with every number traceable to a logged run.

---

## Standing rules for this workstream

1. **No number in a commit message, doc, or PR that was not measured on this
   hardware.** Published results are cited as *others'* results, always.
2. **Losslessness is a gate, not a metric.** Speculative decoding that changes
   output is broken, however fast.
3. **Record the negative results too.** If τ collapses on our merged model, that
   is a finding worth keeping — it tells the next person why.
4. **Verify the premise before building on it.** Two conclusions were already
   overturned in this project by checking an assumption (`X86_AMD_PSTATE=n`
   never applying; "only residency gives 10x"). Cheap to check, expensive to skip.

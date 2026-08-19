# Running Qwen3.6-27B-Fable-Fusion on a two-GPU cluster, and why it stays host-only

**Audience:** an AI engineering student who knows what a transformer is, has run
Ollama, and wants to understand what GreenBoost actually *does* when a 17 GiB
model answers on a 12 GB graphics card, and why "add a second GPU" didn't turn
out to be the win it sounds like.

This is a walkthrough of one real configuration, end to end, including the
dead end:

| | |
|---|---|
| Model | `DavidAU/Qwen3.6-27B-Fable-Fusion-711-...-NEO-MAX-MTP-GGUF` (HF GGUF, `MTP-Q4_K_M`, **17.2 GiB**) |
| Architecture | `qwen35` — **dense**, 27B, every parameter fires on every token. Hybrid: standard attention layers plus **Gated Delta Net** (linear-attention) layers |
| Context | 262,144 tokens native; the box's VRAM currently clamps served context to 2,048 |
| Host | RTX 5070, **12 GB** VRAM · 61 GB DDR5, i9-14900KF (8P+16E) |
| Feeder | RTX 5070 Laptop, **8 GB** VRAM · 29 GB DDR5, over a 1 Gbps LAN |
| Served by | GB-Synapse → `llama-server`, **host-only**, on `:11435` |

The whole problem in one line: **17.2 GiB of weights, ~10-12 GB of usable
single-GPU VRAM.** Unlike the cluster's previous reference model, this one is
**dense, not MoE** — that single fact changes almost every decision below.

---

## 1. Dense changes the placement problem — there is no "cheap tensor" to evict

The previous reference workload was a 256-expert MoE: 92% of its bytes were
experts that only fired 8-per-token, so the fix was surgical — move the
*experts*, keep the *attention*, and pay almost nothing for it
(`--n-cpu-moe`). That trick doesn't exist here. In a dense model every
tensor in every layer fires on every token, so there is no low-traffic
subset to evict cheaply. The only lever left is coarser: **how many whole
layers** sit on the GPU (`-ngl N`) versus the CPU. Every layer moved off the
GPU costs its full share of compute, not just its bytes.

```
weights=17.2GB/10.0GB budget → PARTIAL CPU OFFLOAD (slow): 28/65 layers on GPU
```

This is the honest, structural reason this model is slower on this hardware
than the old MoE reference was: a dense model exceeding VRAM has no free
lunch left to take.

---

## 2. A real bug this model exposed: the compute-graph reserve was too small

The very first load attempt failed outright:

```
resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta Net
  (chunked) is assigned to device CUDA0 (usually due to missing support)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 183.73 MiB on device 0:
  cudaMalloc failed: out of memory
```

GreenBoost's own layer-count planner (`_fit_gpu_layers`) sized `-ngl` against
a flat, %-derived compute-graph reserve tuned for plain attention-only
architectures. This model's hybrid **Gated Delta Net** layers need more
workspace than that flat estimate provides — the weights allocation
succeeded, but a 183 MB compute buffer on top of it didn't. Real evidence,
not a guess: fixed by retrying once at a smaller layer count when this
specific OOM shape is seen (`gb_synapse_backends.py`). Live-verified twice:
both OOM'd at `-ngl 25`, retried and served successfully at `-ngl 20`.

> **Lesson:** a placement budget tuned against one architecture's shape can
> be wrong for a different one, even on the same hardware, even at the same
> nominal VRAM number. The fix is a fast, cheap retry with a real failure
> signature to key off, not a bigger fudge factor guessed in advance.

---

## 3. The RPC-cluster attempt, and the two real bugs it found

The obvious next move: bring the feeder's 8 GB online via `--rpc` and split
the 65 layers across both cards. This surfaced two more real bugs before it
worked at all, and a third reason it isn't used in production even now that
it does.

**Bug: a stale feeder engine binary.** `omen`'s `rpc-server` was 15 days
older than the host's and had no `PINNED_COMMIT` in its vendored
`llama.cpp` checkout — old enough to predate an upstream fix that correctly
excludes discrete GPUs from a Unified-Memory-Architecture memory-reporting
path. The stale binary reported free memory from `/proc/meminfo` (system
RAM) instead of `cudaMemGetInfo` (real VRAM), so the sizing was reading the
wrong number entirely. Fixed by rebuilding the feeder's engine from the
host's pinned commit.

**Bug: llama.cpp's own `-fit` auto-placement conflicts with an explicit
`-ngl`.** Even after the fix above, EVERY layer count tried — 999 down to
11 of 65 — was rejected with the identical message:

```
common_fit_params: failed to fit params to free device memory:
  n_gpu_layers already set by user to N, abort
```

The flat number never mattered. Reading `common/fit.cpp` directly: for a
single device, if the params as given already satisfy the memory margin,
`-fit` returns early with no error at all — that's why solo-host loads
never hit this. For a **multi-device** load, if even one device's margin
isn't already satisfied as given, `-fit` proceeds toward an unconditional
check that throws on ANY explicit `-ngl`, independent of its value.
GreenBoost always sets `-ngl` explicitly — that's the whole point of its
own placement planner — so this fired every single time. Fixed with
`-fit off`: GreenBoost already owns this decision, llama.cpp shouldn't
re-veto it.

> **Lesson:** an identical error message across a wide sweep of otherwise-
> sensible values is a strong signal the *value* isn't the problem — read
> the check itself before tuning the number harder.

**Why it isn't used in production anyway.** With both bugs fixed, placement
correctly climbed to ~48/65 layers across the cluster (up from 20-28/65
solo) and real weight transfer to the feeder was confirmed (its VRAM
climbing as bytes streamed over the LAN). But load-plus-first-decode never
completed cleanly across several patient attempts — a pattern (a 502, then
minutes of 503s after weights had visibly finished loading) that looks like
genuine per-token network round-trip latency: this model's Gated Delta Net
layers on the feeder's side of the split need a cross-node round trip on
*every forward pass*, over a plain 1 Gbps LAN, not a low-latency
interconnect. Owner decision: stay host-only until someone measures RPC
*decode* speed specifically (not just load success) and it turns out
faster. The fixes stay landed regardless — they're correct behavior either
way, this is a policy choice about which placement to serve from, not a
retraction of the fixes.

---

## 4. MTP: still free speed, still provably identical output

This repo ships its own MTP draft head merged directly into the
`MTP`-prefixed quant files — no separate draft model, no extra VRAM.
GB-Synapse detects it from the GGUF's own tensors and enables llama.cpp's
speculative path:

```
--spec-type draft-mtp --spec-draft-n-max 3
```

Live-measured draft acceptance on this model: **64%** (41 of 64 drafted
tokens accepted in one real decode). Speculative decoding drafts cheaply,
the trunk verifies in one batched pass, and rejected drafts are discarded —
the output distribution is the trunk's, exactly, regardless of acceptance
rate. Acceptance rate only affects how much of the "free" speedup you
actually collect.

---

## 5. KV cache and context: clamped hard at first — because of a real bug, not the hardware

This model's native context is **262,144 tokens**. GB-Synapse clamps it down
to fit the live VRAM budget rather than OOMing mid-generation:

```
ctx=262144 would need 32.50 GB KV cache on top of 17.22 GB weights
  (10.01 GB budget) — clamping to ctx=2048
```

That **32.50 GB was wrong**, and it's worth walking through why, because the
correction changed the model from "barely usable" to "actually agentic."
This architecture is hybrid — `16 × (3 × Gated DeltaNet → 1 × Gated
Attention)` — only 1 layer in 4 holds a real, per-token-growing KV cache;
the other 3 (Gated DeltaNet, a linear-attention mechanism) hold a
**fixed-size recurrent state that does not grow with context length at
all**. `estimate_kv_gb()` didn't know this distinction and multiplied by
the full layer count uniformly (65, not the 17 that actually matter),
overestimating KV size by ~3.8x. Confirmed directly from the GGUF's own
`full_attention_interval=4` field, and cross-checked in llama.cpp's own
loader (`is_recr_impl` in `src/models/qwen35.cpp`) — the engine has always
known this internally; GreenBoost's own sizing math just didn't.

Fixed by reading `full_attention_interval` generically and deriving
`ModelEntry.n_kv_layers` from it (1 for a plain transformer — a safe no-op
for every other architecture). **Live result, same hardware, same quant,
same VRAM budget: served ctx went 2048 → 7680, a real 3.75x.** That crossed
a genuine threshold: GB-CLI's own baseline tool/system-prompt overhead
(~4608 tokens) now fits, so a real `gb -p "..."` request can actually think,
call a tool, and read project files — verified live, not assumed. It still
can't complete a full multi-tool-call audit in one turn (real file content
can exceed 7680 tokens on its own), so this isn't "solved," but it's a
different, much better problem than "can't start at all."

`q8_0` KV is used here as a **budget grade**, not a certification grade —
labelled as such in telemetry, same rule as always: a retrieval score is
meaningless without knowing which tier produced it.

**2026-07-29 addendum — the safetensors path had the identical bug, unfixed
until now.** Everything above was fixed on `gb_synapse.gguf_summary()` (the
GGUF/llama.cpp path). The same hybrid architecture served as safetensors
(e.g. `Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP`, NVFP4,
routed through `SynapseTorchBackend`/gLLM instead of llama.cpp) went through
`gb_synapse.safetensors_summary()`, which read the config's `layer_types`
field only for its LENGTH, never its per-layer contents — so `n_kv_layers`
stayed 0 on every hybrid safetensors model, and every consumer's
`entry.n_kv_layers or entry.n_layers` fallback silently treated that as
"assume every layer is real attention," reintroducing the same ~4x
overestimate this section describes fixing for GGUF. Fixed the same day by
counting `layer_types` directly — exact, not an interval approximation,
since the config already lists each layer's type.

That fix also surfaced a second gap that was ALREADY wrong for both GGUF and
safetensors: the recurrent layers' own state (Gated DeltaNet's `conv_state` +
temporal state — small, fixed-size, but per arXiv:2312.00752 read and
rewritten every decode step) was charged as **zero bytes**, everywhere.
`gb_synapse.estimate_ssm_state_gb()` now sizes it (formula ported from
llama.cpp's own `n_embd_r()`/`n_embd_s()`) and `_solve_ctx_and_layers()`
subtracts it from the VRAM budget as a fixed cost alongside weights — never
scaled by ctx, since unlike KV cache it doesn't grow with context length.
The shim can now pin it in T1 (`GREENBOOST_SSM_STATE_MB`, opt-in) instead of
leaving it `GB_ALLOC_ACTIVATIONS`-classified and T2/T3-spill-eligible.

**2026-07-29, later the same day — re-verified live, with one real limit
found.** Metadata: confirmed against the actual
`Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP` config
(`config.json` pulled standalone, no weights) — `n_kv_layers=16` (=
`n_layers/4`, not `0`, not `64`), `n_recurrent_layers=48`,
`ssm_state_gb=0.146`, `quant_method=nvfp4` all detected correctly by
`safetensors_summary()`. `synapse_recommend` on the sibling FP8 checkpoint
(`bottlecapai/ThinkingCap-Qwen3.6-27B-FP8`, identical architecture) confirms
the win live: `kv_gb` 8.0→2.0 GB (exact 4x) with `ssm_state_gb=0.146` now
present where it was always `0` before.

**The end-to-end `synapse_serve` step for THIS checkpoint is blocked, but
not by anything this fix touches**: `gb_synapse.pull()` refuses the NVFP4
repo before downloading — `SynapseTorchBackend`/gLLM has no dispatch path
for `nvfp4` at all (`gllm/layers/linear.py`'s quant dispatch only knows
fp8/gptq/awq), so this is a pre-existing engine-capability gate doing its
job (loud refusal beats a silent bad load), not a Phase-1 regression. See
`workflow/known-issues.md` for the standing gap.

The GGUF path (§5's `LOW-MTP-IQ4_XS` reference numbers) WAS re-served live
end to end: `ctx` moved from the previously-recorded 45824 to **49408**
(host-only, no shim) at `-ngl 33/65` (previously 34/65) — a pre-existing,
documented compute-graph-reserve OOM-retry (`workflow/known-issues.md`,
2026-07-27 entry) fired on the first placement attempt and settled one layer
lower; the freed weight VRAM more than covered the newly-correct 0.146 GB
SSM charge, netting a ctx *increase*, not the "clamps slightly" outcome a
naive read of the fix would predict. `GET /slots` confirmed `n_ctx=49408`
live. The GGUF numbers in this section (2048 → 7680 → ~46K) still do not
carry over exactly to the NVFP4/safetensors checkpoint — different engine,
different placement mechanics — treat them as the GGUF path's own record.

---

## 6. Quality floor: measured, and honestly incomplete

`Q4_K_M` is nominally below fp8. The floor is enforced empirically, not
nominally:

- **`smoke_gate()`**, run 2026-07-27 against the real host-only serving
  path (28/65 layers on GPU, `q8_0` KV, MTP active): **PASS** — no
  repetition collapse (`max6gram=1`, `uniq=1.0`).
- **`niah_certify()`** — **not yet run.** The current `ctx=2048` clamp is
  too small for a meaningful long-context needle test; certifying a number
  at a context this small wouldn't reflect how the model is actually
  reachable at scale. Re-run once ctx headroom improves (more VRAM, a
  working feeder path, or a smaller quant), not before.

This model's own marketing (heavy fine-tune, "uncensored/heretic" merge)
makes measurement more important, not less — a model-card claim is a proxy
for quality, and when proxy and measurement disagree, the measurement wins.

---

## 7. Watching it: dataflux + MCP

Every decision above is real telemetry, queryable live:

```python
dataflux_events(kind="synapse_serve")   # why did it place things this way?
dataflux_events(kind="bench_result")    # the load/decode measurements from this doc
synapse_status()                        # what is actually running right now?
cluster_status()                        # is the feeder online, and is it worth using?
```

The RPC investigation above is a good example of why this matters: the
symptom (identical rejection at every layer count) only made sense once the
underlying `common_fit_params` source was read directly — no amount of
staring at dataflux numbers alone would have found it, but dataflux is what
confirmed the fix actually worked afterward (real weight transfer, no more
spurious rejections) rather than trusting a plausible-sounding theory.

---

## 8. The pipeline, as it actually runs today

```
gb  →  GB-CLI
      │  (OpenAI/Ollama API on :11435)
      ▼
   GB-Synapse proxy
      │
      ▼
   llama-server  (HOST ONLY — see §3 for why the RPC path stays unused)
      ├── -ngl 28                              as many layers as VRAM allows
      ├── --cache-type-k/v q8_0                budget grade (f16 doesn't fit at native ctx)
      ├── --ctx-size 2048                       clamped from 262144 at the time this doc's
      │                                         pipeline snapshot was taken — see §5's own
      │                                         later updates for the fixed, much larger value
      ├── --cache-ram <%-derived from free RAM> host-memory prompt cache — see §9
      ├── --spec-type draft-mtp                 +free decode, ~64% acceptance, identical output
      └── --fit off                             GreenBoost owns placement, not llama.cpp's auto-fit
      │
      ▼
   GB-Dataflux  ──► MCP (synapse_serve, bench_result, smoke_gate, prompt_cache, ...)
```

---

## 9. GB-Semantics governs this model's own metrics now

This model is the reference workload for GB-Semantics' eval fixture
(`tests/fixtures/semantics/events.py`) precisely because it has produced so
many of the raw-field traps that layer exists to catch — the ctx clamp
history in §5 is three consecutive corrections of the same kind of misread
(`n_layers` vs `n_kv_layers`, a budget computed against the wrong weights
figure, a client-side constant that never asked the server). Querying this
model's state should now go through `semantic_resolve`/`semantic_answer`
rather than reading `synapse_status()`/dataflux fields directly:

```python
semantic_resolve("served_ctx")            # the live server's actual ctx, not a guess
semantic_resolve("kv_bits_by_layer")      # k_bits/v_bits + n_kv_layers, hybrid-arch aware
semantic_resolve("tok_s_decode")          # measured, never synapse_recommend()'s estimate
semantic_resolve("prompt_cache_hit_pct")  # new: how well --cache-ram is actually working
semantic_segments("below_quality_floor")  # {matched, evidence} instead of eyeballing bits
```

**The `--cache-ram` addition** (`gb_synapse_backends.py`'s `LlamaCppBackend.
serve()`) targets exactly this model's own agentic use case: `gb`'s terminal
assistant resends a long, stable system prompt + tool schema on every turn,
which is precisely what llama.cpp's host-memory prompt cache is for. Sized
as a percentage of live free host RAM (never a literal MiB figure, per the
hardcoded-hardware-values rule).

**Live-verified 2026-07-29** against this exact model on this exact box
(`--cache-ram 7325 --cache-reuse 256 -np -1`, `LOW-MTP-IQ4_XS`, host-only,
`-ngl 32/65`): two back-to-back `/v1/chat/completions` requests sharing a
system-prompt prefix went from **TTFT ≈1866-2086 ms (cold) to ≈336-354 ms
(warm)** — a ~5-6x drop, `prompt_tokens_details.cached_tokens` confirming
33/37 prompt tokens reused (89.2% hit rate). No cancelled/dropped requests
across 4 real requests, so the `--cache-reuse` + `-np -1` (`kv_unified=true`)
caveat noted upstream did not manifest here — still worth re-checking under
real concurrent multi-slot load, which this single-client test didn't exercise.

This same test surfaced a real bug in the first cut of this telemetry:
GB-CLI's `BACKEND_REGISTRY` talks straight to `/v1/*`
(`openai_passthrough()` in `gb_synapse_api.py`), not the Ollama-compat
routes the `prompt_cache` dataflux kind was originally wired into only — so
the cache-hit measurement was silently blind to all real GB-CLI traffic
until `openai_passthrough()` got the same instrumentation (a side-channel
byte buffer, parsed only after each response is forwarded unchanged, so the
"genuine passthrough" behavior on that route is untouched). A second,
smaller bug in the same fix: the actual field this engine's OpenAI-compat
responses use is `usage.prompt_tokens_details.cached_tokens`, not the
top-level `tokens_cached` llama-server's native `/completion` endpoint uses
— `_cache_info_from_chunk` now checks both shapes. Regression-guarded in
`tests/test_gb_synapse_api_prompt_cache.py`.

Verify the real hit rate via `semantic_resolve("prompt_cache_hit_pct")` /
`dataflux_events(kind="prompt_cache")` — confirmed live to match, not just
assumed from the flag being present.

Nothing here is a heuristic someone liked the sound of, including the parts
that didn't work: the RPC split is real, tested, and its two blocking bugs
are genuinely fixed — it's excluded from production because the *measured*
behavior (a load that never cleanly finished decoding) outweighed the
*theoretical* case for it (more VRAM, more layers on GPU). That is the whole
job: read what the model is, read what the hardware is, measure rather than
assume, and say so plainly when the answer is "not yet."
